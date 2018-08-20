config_lme_hw = {'model':
                     {'formula_fe': 'p_pct_list_hw ~ p_list_hw_log + cost_pct_list_hw'
                      '+ chnl_ep'
                      '+ bundled'
                      '+ date_approval_eoq + upgrade_mes + p_bid_hw_contrib + p_delgl4_hw'
                      '+ p_list_hw_total_log'
                      '+ won'
                    , 'formula_re':
                        '(cost_pct_list_hw||taxon_hw_level_4/taxon_hw_level_3/taxon_hw_level_2/taxon_hw_mtm)'
                        '+ (cost_pct_list_hw||sector/industry)'
                        '+ (cost_pct_list_hw||country)'
                      }
                 , 'FIELD_TYPES':
                     {'p_pct_list_hw': 'float'
                    , 'p_list_hw': 'float'
                    , 'p_list_hw_total': 'float'
                    , 'p_bid_hw': 'float'
                    , 'cost_hw': 'float'
                    , 'p_list_hw_log': 'float'
                    , 'cost_pct_list_hw': 'float'
                    , 'chnl_ep': 'str'
                    , 'bundled': 'float'
                    , 'date_approval_eoq': 'int'
                    , 'upgrade_mes': 'float'
                    , 'p_bid_hw_contrib': 'float'
                    , 'p_delgl4_hw': 'float'
                    , 'p_list_hw_total_log': 'float'
                    , 'won': 'float'
                    , 'taxon_hw_level_4': 'str'
                    , 'taxon_hw_level_3': 'str'
                    , 'taxon_hw_level_2': 'str'
                    , 'taxon_hw_mtm': 'str'
                    , 'sector': 'str'
                    , 'industry': 'str'
                    , 'country': 'str'
                    , 'quote_id': 'str'
                    , 'serial5': 'str'
                    , 'const': 'float'
                    , 'date_approval_year': 'int'
                    , 'component_id': 'str'
                      }
                 , 'ind': ['hw']
                 }

'''
config_lme_hw = {'model':
                     {'formula_fe': 'p_pct_list_hw ~ p_list_hw_log + cost_pct_list_hw'
                      '+ chnl_ep'
                      '+ bundled'
                      '+ date_approval_eoq + upgrade_mes + p_bid_hw_contrib + p_delgl4_hw'
                      '+ p_list_hw_total_log'
                      '+ won'
                    , 'formula_re':
                        # '(p_list_hw_log + cost_pct_list_hw||mtm)'
                        # '(p_list_hw_log +
                        '(cost_pct_list_hw||taxon_hw_level_4/taxon_hw_level_3/taxon_hw_level_2/taxon_hw_mtm)'
                        # +p_list_hw_log
                        '+ (cost_pct_list_hw||sector/industry)'
                        # '+ p_list_hw_log + 
                        '+ (cost_pct_list_hw||country)'
                      }
                 , 'FIELD_TYPES':
                     {'p_pct_list_hw': 'float'
                    , 'p_list_hw_log': 'float'
                    , 'cost_pct_list_hw': 'float'
                    , 'chnl_ep': 'str'
                    , 'bundled': 'float'
                    , 'date_approval_eoq': 'int'
                    , 'upgrade_mes': 'float'
                    , 'p_bid_hw_contrib': 'float'
                    , 'p_delgl4_hw': 'float'
                    , 'p_list_hw_total_log': 'float'
                    , 'won': 'float'
                    , 'taxon_hw_level_4': 'str'
                    , 'taxon_hw_level_3': 'str'
                    , 'taxon_hw_level_2': 'str'
                    , 'taxon_hw_mtm': 'str'
                    , 'sector': 'str'
                    , 'industry': 'str'
                    , 'country': 'str'
                    , 'p_list_hw': 'float'
                    , 'p_list_hw_total': 'float'
                    , 'const': 'float'
                    , 'p_bid_hw': 'float'
                    , 'cost_hw': 'float'
                    }
                 , 'ind': ['hw']
                 }
'''
config_lme_tss = {'model':
                    {'formula_fe': 'p_pct_list_hwma ~ p_list_hwma_log'
                             '+ comp_duration_days + committed'
                             '+ sl_contact_hours + sl_onsite_hours + sl_part_hours + sl_hours_x_days'
                             '+ tss_bundled'
                             '+ tss_type + upgrade_mes'
                             '+ chnl_tss + p_list_hwma_hw_log'
                    , 'formula_re': '(p_list_hwma_log||taxon_hw_level_4/taxon_hw_level_3/taxon_hw_level_2/mtm)'
                               '+ (p_list_hwma_log||sector/industry)'
                               '+ (p_list_hwma_log||market/country)'
                    }
                , 'FIELD_TYPES':
                     {'p_pct_list_hwma': 'float'
                    , 'p_list_hwma': 'float'
                    , 'p_list_hwma_log': 'float'
                    , 'comp_duration_days': 'int'
                    , 'committed': 'float'
                    , 'sl_contact_hours': 'float'
                    , 'sl_onsite_hours': 'float'
                    , 'sl_part_hours': 'float'
                    , 'sl_hours_x_days': 'float'
                    , 'tss_bundled': ' int'
                    , 'tss_type': 'str'
                    , 'upgrade_mes': 'float'
                    , 'chnl_tss': 'str'
                    , 'p_list_hwma_hw_log': 'float'
                    , 'taxon_hw_level_4': 'str'
                    , 'taxon_hw_level_3': 'str'
                    , 'taxon_hw_level_2': 'str'
                    , 'mtm': 'str'
                    , 'sector': 'str'
                    , 'industry': 'str'
                    , 'market': 'str'
                    , 'country': 'str'
                    , 'p_bid': 'float'
                    , 'serial5': 'str'
                    , 'date_inst': 'str'
                    , 'date_approval_year': 'int'
                    , 'const': 'float'
                    , 'cost': 'float'
                    , 'PTI_0price': 'float'
                    , 'p_bid_hwma': 'float'
                      }
                , 'ind': ['tss']
          }

config_wr_hw = {'quant_feats': {'in_feats': ['const', 'p_list_hw_log'], 'target': 'metric'
                                    }
                    , 'ind': ['hw']
                    , 'index': ['index']
                    , 'grp_cols': ['taxon_hw_level_3']
                    , 'percentile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    , 'labels':
                        {
                            '0.1': 'L'
                            , '0.2': 'L1'
                            , '0.3': 'L2'
                            , '0.4': 'L3'
                            , '0.5': 'M'
                            , '0.6': 'H3'
                            , '0.7': 'H2'
                            , '0.8': 'H1'
                            , '0.9': 'H'
                        }
          }

config_wr_tss = {'quant_feats': {'in_feats': ['const', 'p_list_hwma_log', 'p_list_hwma_hw_log'], 'target': 'metric'
                                    }
                    , 'ind': ['tss']
                    , 'index': ['index']
                    , 'grp_cols': ['taxon_hw_level_3']
                    , 'percentile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    , 'labels':
                        {
                            '0.1': 'L'
                            , '0.2': 'L1'
                            , '0.3': 'L2'
                            , '0.4': 'L3'
                            , '0.5': 'M'
                            , '0.6': 'H3'
                            , '0.7': 'H2'
                            , '0.8': 'H1'
                            , '0.9': 'H'
                        }
          }

config_lme_tsso = {'model':
                    {'formula_fe': 'p_pct_list_hwma ~ p_list_hwma_log + costfactor'
                             '+ committed'
                             '+ sl_contact_hours + sl_onsite_hours + sl_part_hours + sl_hours_x_days'
                             '+ tss_bundled'
                             '+ tss_type'
                             '+ chnl_tss'
                    , 'formula_re': '(p_list_hwma_log||taxon_hw_level_4/taxon_hw_level_3/taxon_hw_level_2/mtm)'
                               '+ (p_list_hwma_log||market/country)'
                    }
                , 'FIELD_TYPES':
                     {'p_pct_list_hwma': 'float'
                    , 'p_list_hwma': 'float'
                    , 'p_list_hwma_log': 'float'
                    , 'committed': 'float'
                    , 'sl_contact_hours': 'float'
                    , 'sl_onsite_hours': 'float'
                    , 'sl_part_hours': 'float'
                    , 'sl_hours_x_days': 'float'
                    , 'tss_bundled': 'float'
                    , 'tss_type': 'str'
                    , 'chnl_tss': 'str'
                    , 'taxon_hw_level_4': 'str'
                    , 'taxon_hw_level_3': 'str'
                    , 'taxon_hw_level_2': 'str'
                    , 'mtm': 'str'
                    , 'market': 'str'
                    , 'country': 'str'
                    , 'p_bid': 'float'
                    , 'serial5': 'str'
                    , 'date_inst': 'str'
                    , 'const': 'float'
                    , 'cost': 'float'
                    , 'costfactor': 'float'
                    , 'PTI_0price': 'float'
                    , 'p_bid_hwma': 'float'
                    , 'contract_number': 'str'
                      }
                , 'ind': ['tsso']
          }


config_wr_tsso = {'quant_feats': {'in_feats': ['const', 'p_list_hwma_log'], 'target': 'metric'
                                    }
                    , 'ind': ['tsso']
                    , 'index': ['index']
                    , 'grp_cols': ['taxon_hw_level_3']
                    , 'percentile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    , 'labels':
                        {
                            '0.1': 'L'
                            , '0.2': 'L1'
                            , '0.3': 'L2'
                            , '0.4': 'L3'
                            , '0.5': 'M'
                            , '0.6': 'H3'
                            , '0.7': 'H2'
                            , '0.8': 'H1'
                            , '0.9': 'H'
                        }
          }

FIELD_MAP = {'input_trans': {
    'ComListPrice': 'p_list_hw'
    , 'ComLogListPrice': 'p_list_hw_log'
    , 'ComPctContrib': 'p_bid_hw_contrib'
    , 'Componentid': 'componentid'
    , 'ComTMC': 'cost_hw'
    , 'ComCostPofL': 'cost_pct_list_hw'
    , 'ComQuotePrice': 'p_bid_hw'
    , 'EndOfQtr': 'date_approval_eoq'
    , 'UpgMES': 'upgrade_mes'
    , 'ComDelgPriceL4': 'p_delgl4_hw'
    , 'WinLoss': 'won'
    , 'Level_4': 'taxon_hw_level_4'
    , 'Level_3': 'taxon_hw_level_3'
    , 'Level_2': 'taxon_hw_level_2'
    , 'Level_1': 'taxon_hw_mtm'
    , 'ctry_desc': 'country'
    , 'totalcharge': 'p_list_hwma'
    , 'basecharge': 'p_bid_hwma'
    , 'PTI0': 'PTI_0price'
    , 'PTI5': 'PTI_5price'
    , 'Cost': 'cost'
    , 'TSSContduration': 'comp_duration_days'
    , 'coverage_hours_days': 'sl_hours_x_days'
    , 'sl_cntct': 'sl_contact_hours'
    , 'sl_onsite': 'sl_onsite_hours'
    , 'sl_part_time': 'sl_part_hours'
    , 'LogDealSize': 'p_list_hw_total_log'
    , 'imt_code': 'market'
}
    , 'hw_cols': ['QuoteID', 'index', 'componentid', 'p_list_hw', 'p_list_hw_log', 'cost_hw', 'cost_pct_list_hw',
                  'p_bid_hw_contrib', 'DealSize', 'p_list_hw_total_log', 'TSSComponentincluded',
                  'Indirect(1/0)', 'date_approval_eoq', 'upgrade_mes',
                  'p_delgl4_hw', 'won', 'taxon_hw_level_4', 'taxon_hw_level_3', 'taxon_hw_level_2', 'taxon_hw_mtm',
                  'CustomerSecName', 'CustomerIndustryName', 'country', 'p_bid_hw'
                  ]

    , 'new_hw_cols': ['p_bid_hw_contrib', 'p_list_hw_total_log', 'taxon_hw_level_4', 'taxon_hw_level_3',
                      'taxon_hw_level_2', 'taxon_hw_mtm', 'sector', 'industry', 'bundled', 'chnl_ep', 'const',
                      ]

    , 'tss_cols': ['QuoteID', 'index', 'componentid', 'p_list_hwma', 'p_list_hw', 'PTI_0price', 'PTI_5price', 'cost', 'Indirect(1/0)',
                   'TSSComponentincluded', 'upgrade_mes', 'committedcharge',
                   'comp_duration_days', 'sl_hours_x_days', 'sl_contact_hours', 'sl_onsite_hours', 'sl_part_hours',
                   'servoffcode', 'taxon_hw_level_4', 'taxon_hw_level_3', 'taxon_hw_level_2', 'taxon_hw_mtm',
                   'CustomerSecName', 'CustomerIndustryName', 'market', 'country', 'p_bid_hwma'
                   ]

    , 'new_tss_cols': ['p_list_hwma_log', 'p_list_hwma_hw_log', 'committed', 'chnl_tss',
                       'mtm_tss', 'tss_type', 'tss_bundled', 'const'
                       ]

    , 'common_cols': ['QuoteID', 'index', 'Componentid', 'Indirect(1/0)', 'CustomerSecName', 'CustomerIndustryName',
                      'country']

    , 'initial_cols': ['CustomerNumber', 'ClientSegCd', 'ClientSeg=E', 'CCMScustomerNumber',
                       'CustomerSecName', 'CustomerIndustryName', 'RequestingApplicationID', 'ModelID', 'Version',
                       'Countrycode', 'ChannelID', 'Year', 'Month', 'EndOfQtr', 'Indirect(1/0)', 'DomBuyerGrpID',
                       'DomBuyerGrpName', 'QuoteType', 'TSSComponentincluded', 'TSSContstartdate',
                       'TSSContenddate', 'TSSContduration', 'TSSPricerefdate', 'TSSPricoptdescription',
                       'TSSFrameOffering', 'Componentid', 'Quantity', 'UpgMES', 'ComListPrice', 'ComTMC',
                       'ComQuotePrice', 'ComDelgPriceL4', 'ComRevCat', 'ComRevDivCd', 'ComBrand', 'ComGroup',
                       'ComFamily', 'ComMT', 'ComMTM', 'ComQuotePricePofL', 'ComDelgPriceL4PofL', 'ComCostPofL',
                       'ComLogListPrice', 'ComMTMDesc', 'ComMTMDescLocal', 'ComCategory', 'ComSubCategory',
                       'ComSpclBidCode1', 'ComSpclBidCode2', 'ComSpclBidCode3', 'ComSpclBidCode4', 'ComSpclBidCode5',
                       'ComSpclBidCode6', 'HWPlatformid',
                       'FeatureId', 'FeatureQuantity', 'Level_1', 'Level_2', 'Level_3', 'Level_4', 'TSScomid',
                       'type', 'model', 'TSS_quantity', 'warranty', 'warrantyperiod', 'servoffdesc', 'servoffcode',
                       'servstartdate', 'servenddate', 'serviceduration', 'inststartdate', 'instenddate', 'serial',
                       'servlvldesc', 'servlvlcode', 'basecharge', 'committedcharge', 'totalcharge', 'PTI0',
                       'CMDAprice', 'Cost', 'imt_code', 'coverage_hours_days', 'coverage_hours',
                       'coverage_days', 'sl_cntct', 'sl_fix_time', 'sl_onsite', 'sl_part_time', 'p_uplift_comm',
                       'TssComponentType', 'hwma_pti',
                       'swma_discounted_price', 'ParentMapping_ComponentID', 'QuoteID', 'WinLoss', 'ComLowPofL',
                       'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize', 'LogDealSize', 'ComPctContrib',
                       'ComRevDivCd_Orig'
                       ]  # 'hw_level1', 'hw_level2', 'hw_level3', 'hw_level4', 'flex', 'ecs',
    , 'final_cols': ['CustomerNumber', 'ClientSegCd', 'ClientSeg=E', 'CCMScustomerNumber',
                     'CustomerSecName', 'CustomerIndustryName', 'RequestingApplicationID', 'ModelID', 'Version',
                     'Countrycode', 'ChannelID', 'Year', 'Month', 'EndOfQtr', 'Indirect(1/0)', 'DomBuyerGrpID',
                     'DomBuyerGrpName', 'QuoteType', 'TSSComponentincluded', 'Quantity', 'UpgMES',
                     'ComListPrice', 'ComTMC', 'ComQuotePrice', 'ComDelgPriceL4', 'ComRevCat', 'ComRevDivCd',
                     'ComBrand', 'ComGroup', 'ComFamily', 'ComMT', 'ComMTM', 'ComQuotePricePofL',
                     'ComDelgPriceL4PofL', 'ComCostPofL', 'ComLogListPrice', 'ComMTMDesc', 'ComMTMDescLocal',
                     'ComCategory', 'ComSubCategory', 'ComSpclBidCode1', 'ComSpclBidCode2', 'ComSpclBidCode3',
                     'ComSpclBidCode4', 'ComSpclBidCode5', 'ComSpclBidCode6', 'HWPlatformid', 'FeatureId',
                     'FeatureQuantity', 'Level_1', 'Level_2', 'Level_3', 'Level_4', 'TSScomid', 'type', 'model',
                     'TSS_quantity',
                     'warranty', 'warrantyperiod', 'servoffdesc', 'servoffcode', 'servstartdate', 'servenddate',
                     'serviceduration', 'inststartdate', 'instenddate', 'serial', 'servlvldesc', 'servlvlcode',
                     'basecharge', 'committedcharge', 'totalcharge', 'PTI0', 'CMDAprice', 'Cost', 'QuoteID',
                     'WinLoss', 'ComLowPofL', 'ComMedPofL', 'ComHighPofL', 'ComMedPrice', 'DealSize',
                     'LogDealSize', 'ComPctContrib', 'ComRevDivCd_Orig', 'Componentid', 'coverage_hours_days',
                     'coverage_hours', 'coverage_days', 'sl_cntct', 'sl_fix_time',
                     'sl_onsite', 'sl_part_time', 'p_uplift_comm', 'TSSContstartdate', 'TSSContenddate',
                     'TSSContduration', 'TSSPricerefdate', 'TSSPricoptdescription', 'TSSFrameOffering',
                     'ParentMapping_ComponentID', 'GEO_CODE', 'TreeNode', 'DealBotLineSpreadOptimalPrice',
                     'OptimalPrice', 'OptimalPriceExpectedGP', 'OptimalPriceGP', 'OptimalPriceIntervalHigh',
                     'OptimalPriceIntervalLow', 'OptimalPricePofL', 'OptimalPriceWinProb', 'PredictedQuotePrice',
                     'PredictedQuotePricePofL', 'QuotePrice', 'QuotePriceExpectedGP', 'QuotePriceGP',
                     'QuotePricePofL', 'QuotePriceWinProb', 'AdjComHighPofL', 'AdjComHighPrice', 'AdjComLowPofL',
                     'AdjComLowPrice', 'AdjComMedPofL', 'AdjComMedPrice', 'ComLowPrice', 'ComHighPrice',
                     'TSS_DealBotLineSpreadOptimalPrice', 'TSS_OptimalPrice', 'TSS_OptimalPriceExpectedGP',
                     'TSS_OptimalPriceGP', 'TSS_OptimalPriceIntervalHigh', 'TSS_OptimalPriceIntervalLow',
                     'TSS_OptimalPricePofL', 'TSS_OptimalPriceWinProb', 'TSS_AdjComHighPofL',
                     'TSS_AdjComHighPrice', 'TSS_AdjComLowPofL', 'TSS_AdjComLowPrice', 'TSS_AdjComMedPofL',
                     'TSS_AdjComMedPrice', 'TssComponentType', 'hwma_pti', 'pti', 'swma_discounted_price'
                     ]  # removed 'flex', 'ecs',

}